#!/usr/bin/env bash
# Unit tests for scripts/sycl-decode-mode-capture.sh, against a fake sysfs
# tree and a fake bench command -- mirrors tests/test-bench-guard.sh's
# mk_tree/mk_meminfo/expect_status pattern. All host-corruption preflight
# (throttled/active card, stale tenant, elevated Shmem) belongs to
# bench-guard.sh, which the capture script invokes as a child and which has
# its own test suite (test-bench-guard.sh); this file exercises only the
# capture script itself: the sampler timeline, host.txt before/after blocks,
# tg128 parsing + mode computation, and refusal pass-through.
#
# Expected wall-clock runtime: roughly 30-45s (varies with host load -- see
# CLAUDE.md's "host load is permanent" note). Dominated by
# mk_fake_bench_grow's fixed ~2s busy-wait in three separate cases
# (pid-check, busy-card, helper-race), mk_tree_busy_then_free's ~10s
# worst-case poll-recovery window in the busy-card case, and
# MK_FAKE_BENCH_SLOW_SECONDS in the one case that asserts a resolved
# bench_pid (three-tick confirmation needs real ticks to accumulate, and
# this suite's own per-tick subprocess overhead under load can push actual
# tick spacing well past the script's nominal cadence -- see
# MK_FAKE_BENCH_SLOW_SECONDS's own comment below). No fixed TIMEOUT is set
# on this test's tests/CMakeLists.txt registration, unlike the 30-120s
# TIMEOUTs some neighbouring registrations carry: these fixtures are
# load-sensitive by design, so a timeout tuned for a quiet host would flake
# under load exactly the way one fixture already did before its own margin
# was widened (llama.cpp-gvu7 quality review).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAPTURE="$ROOT_DIR/scripts/sycl-decode-mode-capture.sh"
GUARD="$ROOT_DIR/scripts/bench-guard.sh"
# CAPTURE ships in the SAME commit as this test (scripts/sycl-decode-mode-
# capture.sh and this file were never landed separately, unlike the RED
# phase before the script existed at all) -- its absence here is a defect
# (a lost mode bit, a bad path, a botched merge), not a legitimate "not
# built yet" skip, so this fails closed rather than exiting 77
# (llama.cpp-gvu7 review, round 2, nit 4). bench-guard.sh is a pre-existing,
# separately-shipped script this test invokes as a child; a checkout
# missing it is a different, genuinely skippable situation.
[ -x "$CAPTURE" ] || { echo "FAIL: $CAPTURE is missing or not executable -- it ships in the same commit as this test, so its absence is a defect" >&2; exit 1; }
[ -x "$GUARD" ] || { echo "SKIP: bench-guard.sh not present"; exit 77; }

# CONFIRM_TICKS/TICK_SECONDS: read directly out of $CAPTURE with `sed`
# instead of a hand-maintained mirror. An earlier version of this file
# declared its own CONFIRM_TICKS=3/TICK_SECONDS=0.5 "mirrors" and claimed a
# script-side edit that widened either without updating them would "fail
# loudly" -- that claim was false in the direction that matters: the sanity
# guard below is computed from the TEST's own copies, so it cannot see a
# script-side change at all (demonstrated: bumping the script's
# CONFIRM_TICKS from 3 to 9 with the mirror left at 3 produced no guard
# failure, only a confusing downstream "bench_pid=unknown" further down --
# llama.cpp-gvu7 quality review round 2). Reading the values FROM the
# script closes that gap structurally: there is nothing left to drift out
# of sync. A rename or reformat of either assignment line in the script
# makes the sed pattern match nothing, and the explicit `-n` check below
# fails this test closed (exit 1) rather than silently computing a
# margin-of-zero from an empty string.
CONFIRM_TICKS="$(sed -n 's/^CONFIRM_TICKS=\([0-9][0-9]*\).*/\1/p' "$CAPTURE")"
TICK_SECONDS="$(sed -n 's/^TICK_SECONDS=\([0-9.][0-9.]*\).*/\1/p' "$CAPTURE")"
[ -n "$CONFIRM_TICKS" ] || { echo "FAIL: could not read CONFIRM_TICKS out of $CAPTURE (renamed or reformatted?)" >&2; exit 1; }
[ -n "$TICK_SECONDS" ] || { echo "FAIL: could not read TICK_SECONDS out of $CAPTURE (renamed or reformatted?)" >&2; exit 1; }

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

# MK_FAKE_BENCH_SLOW_SECONDS: mk_fake_bench's default sleep duration, for
# the one case (out_slow) that asserts a resolved bench_pid. pid commitment
# needs CONFIRM_TICKS consecutive TICK_SECONDS-spaced ticks to land on the
# SAME resolved pid (llama.cpp-gvu7 review, round 2) -- a bare minimum of
# (CONFIRM_TICKS-1)*TICK_SECONDS = 1.0s of real ticks (N ticks spaced
# TICK_SECONDS apart span (N-1)*TICK_SECONDS, not N*TICK_SECONDS -- see
# confirm_bench_pid's own comment in the script) -- and this suite's own
# per-tick overhead (several subprocess forks per sample in the capture
# script) under this host's permanent ambient load can push the EFFECTIVE
# tick spacing well past TICK_SECONDS -- caught directly by a flake loop
# when this fixture used a flat `sleep 2`: too little margin, and the
# "bench_pid resolved" check below intermittently saw "unknown" (2/10 runs,
# all with 3+ real ticks simply not fitting in 2s). 4s is ample margin even
# under load. mk_fake_bench_grow's own busy-wait is not similarly extended
# because its scripts stay leaf from the moment they start (no forked child
# to hand tracking off to), so they warm up faster in practice -- but if
# that ever flakes too, the fix is the same: more real time, not fewer
# required ticks.
MK_FAKE_BENCH_SLOW_SECONDS=4
# MK_FAKE_BENCH_FAST_SECONDS: for every OTHER mk_fake_bench case, which
# asserts only mode/tg128/exit-status, never bench_pid -- these don't need
# three ticks to elapse at all, so a short, fixed duration keeps the suite's
# total runtime down without weakening any assertion (quality review,
# finding 8's "cheap partial" suggestion).
MK_FAKE_BENCH_FAST_SECONDS=1

# Sanity guard, derived from the CONFIRM_TICKS/TICK_SECONDS values just read
# out of the script (above) rather than a second hand-picked number:
# MK_FAKE_BENCH_SLOW_SECONDS must clear the tick-based floor with real
# margin, or the "bench_pid resolved" assertions below are set up to flake
# the way this fixture already did once at `sleep 2`. The floor itself is
# (CONFIRM_TICKS-1)*TICK_SECONDS, not CONFIRM_TICKS*TICK_SECONDS -- N ticks
# spaced TICK_SECONDS apart span (N-1)*TICK_SECONDS (llama.cpp-gvu7 quality
# review round 2, item 3). Because CONFIRM_TICKS/TICK_SECONDS are now read
# live from $CAPTURE instead of hand-mirrored, this guard also fails loudly
# (rather than silently computing a stale floor) if the script's tunables
# change without this fixture's margin being revisited.
MIN_CONFIRM_SECONDS="$(awk -v t="$CONFIRM_TICKS" -v s="$TICK_SECONDS" 'BEGIN { print (t - 1) * s }')"
awk -v have="$MK_FAKE_BENCH_SLOW_SECONDS" -v min="$MIN_CONFIRM_SECONDS" 'BEGIN { exit !(have >= min * 2) }' \
    || { echo "FAIL: MK_FAKE_BENCH_SLOW_SECONDS ($MK_FAKE_BENCH_SLOW_SECONDS) has too little margin over 2x (CONFIRM_TICKS-1)*TICK_SECONDS ($MIN_CONFIRM_SECONDS)"; fail=1; }

# mk_fake_bench: a fake "llama-bench" that prints a canned markdown tg128 row
# embedding $1 as the t/s figure, then sleeps $2 seconds (default
# MK_FAKE_BENCH_SLOW_SECONDS; an EXTERNAL `sleep`, so this script forks a
# child partway through its life) so the sampler gets several samples. The
# capture script's pid discovery is pure structural leaf descent (no
# name/comm matching at all -- see find_bench_pid's comment in the script),
# so once this script forks `sleep` the leaf being tracked moves from this
# script to that child; that's fine for the tests below that only need SOME
# pid to be resolved (mode/tg128 parsing, timeline shape, host.txt shape).
# It is deliberately NOT used for the pid/RSS regression check further
# down, which needs the tracked leaf to stay this script's own process
# throughout -- see mk_fake_bench_grow.
mk_fake_bench() {
    local tg="$1" seconds="${2:-$MK_FAKE_BENCH_SLOW_SECONDS}" path="$T/fakebench.sh"
    cat > "$path" <<EOF
#!/usr/bin/env bash
echo '| model | size | params | backend | ngl | test | t/s |'
echo '|---|---|---|---|---|---|---|'
echo "| gpt-oss 20B MXFP4 | 12.83 GiB | 20.91 B | SYCL | 99 | tg128 | $tg ± 0.31 |"
sleep $seconds
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

# --- live --drm-root derivation (llama.cpp-o4fs) -------------------------
#
# Before this fix, $CAPTURE carried its own PRIVATE selector->PCI table
# (level_zero:0 -> 0000:03:00.0, level_zero:1 -> 0000:07:00.0) instead of
# sharing bench-guard.sh's derive_card_for_selector, and had no --drm-root
# flag at all -- every case below would have failed against that code:
# --drm-root would hit the "unknown arg" branch (exit 2), and even a
# variant using the real /sys/class/drm path could never exercise a
# card-order-!=-PCI-order fixture the way bench-guard.sh's own suite does,
# since the private table only ever produced the two hardcoded, now-stale
# addresses. The derivation arm was therefore dead code as far as this
# suite could tell: every case above passes --sysfs-card explicitly, which
# always bypasses derivation entirely.
#
# mk_pci_dev/mk_drmroot mirror tests/test-bench-guard.sh's own fixture
# builders of the same name exactly (including the deliberate
# card-order-!=-PCI-order + connector-entry shape) -- kept as a local copy
# rather than shared, the same way this file already duplicates
# mk_tree/mk_meminfo/expect_status from that suite, per this file's own
# header comment.
mk_pci_dev() {
    local devroot="$1" addr="$2" with_freq="${3:-}"
    mkdir -p "$devroot/$addr"
    echo 0x8086 > "$devroot/$addr/vendor"
    echo 0x030000 > "$devroot/$addr/class"
    if [ -n "$with_freq" ]; then
        mkdir -p "$devroot/$addr/tile0/gt0/freq0/throttle"
        echo 0 > "$devroot/$addr/tile0/gt0/freq0/throttle/status"
        echo 0 > "$devroot/$addr/tile0/gt0/freq0/act_freq"
    fi
}

mk_drmroot() {
    local d="$T/drmroot" devroot="$T/devices-lz01"
    rm -rf "$d" "$devroot"
    mk_pci_dev "$devroot" 0000:09:00.0 with_freq
    mk_pci_dev "$devroot" 0000:00:02.0
    mk_pci_dev "$devroot" 0000:04:00.0 with_freq
    mkdir -p "$d/card0" "$d/card1" "$d/card2" "$d/card0-DP-1"
    ln -s "$devroot/0000:09:00.0" "$d/card0/device"
    ln -s "$devroot/0000:00:02.0" "$d/card1/device"
    ln -s "$devroot/0000:04:00.0" "$d/card2/device"
    ln -s "$devroot/0000:04:00.0" "$d/card0-DP-1/device"
}

mk_drmroot; mk_meminfo 3000000
out_lz0="$T/out-lz0"
bench="$(mk_fake_bench 40.0 "$MK_FAKE_BENCH_FAST_SECONDS")"
env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$CAPTURE" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --journalctl-cmd true --max-wait 1 \
    --out "$out_lz0" -- "$bench" || { echo "FAIL: level_zero:0 --drm-root run failed"; fail=1; }
head -1 "$out_lz0/bench.log" | grep -q "card=$T/drmroot/card2" \
    || { echo "FAIL: level_zero:0 must derive card=$T/drmroot/card2 (lower PCI 0000:04:00.0; card0->09, card2->04), not the card0-DP-1 connector (got: $(head -1 "$out_lz0/bench.log" 2>/dev/null))"; fail=1; }

mk_meminfo 3000000
out_lz1="$T/out-lz1"
bench="$(mk_fake_bench 40.0 "$MK_FAKE_BENCH_FAST_SECONDS")"
env ONEAPI_DEVICE_SELECTOR=level_zero:1 "$CAPTURE" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --journalctl-cmd true --max-wait 1 \
    --out "$out_lz1" -- "$bench" || { echo "FAIL: level_zero:1 --drm-root run failed"; fail=1; }
head -1 "$out_lz1/bench.log" | grep -q "card=$T/drmroot/card0" \
    || { echo "FAIL: level_zero:1 must derive card=$T/drmroot/card0 (higher PCI 0000:09:00.0) (got: $(head -1 "$out_lz1/bench.log" 2>/dev/null))"; fail=1; }

# Out-of-range: only two discrete cards in mk_drmroot's fixture, so
# level_zero:2 must refuse with exit 3 -- and must not even reach $OUT
# (setup-only failure, same rule the pre-existing setup-fail-regression
# case below enforces for the no-selector case).
out_lz2="$T/out-lz2"
bench="$(mk_fake_bench 40.0 "$MK_FAKE_BENCH_FAST_SECONDS")"
lz2_rc=0
out_lz2_text="$(env ONEAPI_DEVICE_SELECTOR=level_zero:2 "$CAPTURE" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --journalctl-cmd true --max-wait 1 \
    --out "$out_lz2" -- "$bench" 2>&1)" || lz2_rc=$?
[ "$lz2_rc" -eq 3 ] || { echo "FAIL: level_zero:2 (out of range) must refuse with exit 3, got $lz2_rc (out: $out_lz2_text)"; fail=1; }
echo "$out_lz2_text" | grep -qi "out of range" \
    || { echo "FAIL: out-of-range refusal must say so (got: $out_lz2_text)"; fail=1; }
[ ! -e "$out_lz2" ] || { echo "FAIL: an out-of-range setup failure must not create --out at all"; fail=1; }

# --- M1 (spec review round 2): scripts/sycl-decode-mode-capture.sh's own
# explicit `DRM_ROOT=/sys/class/drm` (set unconditionally before sourcing
# sycl-gpu-sysfs.sh, mirroring bench-guard.sh's own F1 fix) had no test
# coverage of its own. As with bench-guard.sh's own F1 regression test
# (tests/test-bench-guard.sh), this cannot be exercised by ALSO passing
# --drm-root (that flag's arg-parser assignment always wins regardless of
# the fix) or --sysfs-card (that bypasses derivation entirely) -- the only
# way to reach the real code path is to omit both and let full derivation
# run, with env DRM_ROOT pointed at a decoy tree. A DECOY drm root (a
# single fake card at a made-up address, 0000:55:00.0) is pointed to by
# env DRM_ROOT; this run must never resolve against it -- whatever it
# resolves against instead (this host's real /sys/class/drm, or a
# refusal) is acceptable, since only leaking the decoy through is what
# this guards against.
rm -rf "$T/drmroot-decoy" "$T/devices-decoy"
mk_pci_dev "$T/devices-decoy" 0000:55:00.0 with_freq
mkdir -p "$T/drmroot-decoy/card0"
ln -s "$T/devices-decoy/0000:55:00.0" "$T/drmroot-decoy/card0/device"
out_m1="$T/out-m1-drmroot-env"
mk_meminfo 3000000
bench="$(mk_fake_bench 40.0 "$MK_FAKE_BENCH_FAST_SECONDS")"
m1_rc=0
m1_out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:0 DRM_ROOT="$T/drmroot-decoy" "$CAPTURE" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --journalctl-cmd true --max-wait 1 \
    --out "$out_m1" -- "$bench" 2>&1)" || m1_rc=$?
if echo "$m1_out" | grep -q "drmroot-decoy"; then
    echo "FAIL: an inherited env DRM_ROOT must not be honoured -- the decoy tree leaked into the derivation (rc=$m1_rc, out: $m1_out)"
    fail=1
fi
if [ -f "$out_m1/bench.log" ] && grep -q "drmroot-decoy" "$out_m1/bench.log"; then
    echo "FAIL: bench.log must not reference the decoy drm root"
    fail=1
fi
if [ "$m1_rc" -ne 0 ] && [ "$m1_rc" -ne 3 ]; then
    echo "FAIL: expected rc 0 (resolved against the real tree) or 3 (real tree has no usable discrete GPU / other refusal), got $m1_rc (out: $m1_out)"
    fail=1
fi

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
bench="$(mk_fake_bench 39.5 "$MK_FAKE_BENCH_FAST_SECONDS")"
run_capture "$out_fast" -- "$bench" || { echo "FAIL: fast-mode run failed"; fail=1; }
grep -q "mode=fast" "$out_fast/mode.txt" 2>/dev/null || { echo "FAIL: expected mode=fast (got: $(cat "$out_fast/mode.txt" 2>/dev/null))"; fail=1; }

mk_tree 0 0; mk_meminfo 3000000
out_mid="$T/out-unknown"
bench="$(mk_fake_bench 34.0 "$MK_FAKE_BENCH_FAST_SECONDS")"
run_capture "$out_mid" -- "$bench" || { echo "FAIL: mid-band run failed"; fail=1; }
grep -q "mode=unknown" "$out_mid/mode.txt" 2>/dev/null || { echo "FAIL: expected mode=unknown (got: $(cat "$out_mid/mode.txt" 2>/dev/null))"; fail=1; }

# --- timeline.tsv: >= 3 sampled rows over the fake bench's 2s sleep ---

[ -s "$out_slow/timeline.tsv" ] || { echo "FAIL: timeline.tsv missing/empty"; fail=1; }
rows=$(( $(wc -l < "$out_slow/timeline.tsv") - 1 ))
[ "$rows" -ge 3 ] || { echo "FAIL: expected >= 3 timeline data rows, got $rows"; fail=1; }
head -1 "$out_slow/timeline.tsv" | grep -qE '^t_s	act_freq	cur_freq	throttle_status	reason_pl2	bench_pid	rss_anon_kb$' \
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
# it with roughly 2x margin. Column 7 (rss_anon_kb) -- column 6 is now the
# bench_pid the review added.
max_rss="$(awk -F'\t' 'NR>1 && $7 != "-" { v = $7 + 0; if (v > max) max = v } END { print max + 0 }' "$out_pid/timeline.tsv")"
[ "$max_rss" -gt 32768 ] \
    || { echo "FAIL: expected a sampled RssAnon > 32768 kB (fixture holds ~64 MiB); got max=$max_rss kB -- RssAnon may be tracking a wrapper, not the bench"; fail=1; }

# --- regression: proof B (llama.cpp-gvu7 review, rounds 2 and 3). While
# the card is (simulated) busy, bench-guard's OWN preflight throttle/tenant
# poll loop forks its own `sleep 5` directly under the guard -- structural
# leaf descent WILL transiently latch onto that (expected: it is genuinely
# the deepest live process at that instant; `sleep` easily survives the
# two-tick confirmation below too, since it runs for the full 5s poll
# interval = 10 ticks, far more than two). What must NOT happen: those
# pre-bench rows must never be attributed to "the bench" -- the review's
# round-2 ruling on the header's own claim ("RssAnon of the bench pid").
# mk_tree_busy_then_free starts throttle/status=1 and flips it to 0 via a
# background subshell after ~6s, mirroring the reviewer's own repro;
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

# Row-level attribution (column 6 is bench_pid, column 7 is rss_anon_kb):
# every row while bench_pid reads "-" (the pre-bench/poll-sleep phase) must
# ALSO read "-" for RssAnon -- never the poll sleep's ~1.2 MB attributed to
# "the bench" -- and every row that DOES carry a bench_pid must carry the
# fixture's own real pid, never a wrapper's.
bad_attribution="$(awk -F'\t' -v want="$want_busy_pid" '
    NR>1 {
        if ($6 == "-") { if ($7 != "-") print "pre-bench row has a non-\"-\" RssAnon: " $0 }
        else if ($6 != want) { print "row bench_pid is not the fixture'"'"'s pid: " $0 }
        else { seen_real = 1 }
    }
    END { if (!seen_real) print "no row ever carried the real bench pid" }
' "$out_busy/timeline.tsv")"
[ -z "$bad_attribution" ] \
    || { echo "FAIL: busy-card timeline row attribution wrong:"; echo "$bad_attribution"; fail=1; }

max_rss_busy="$(awk -F'\t' 'NR>1 && $7 != "-" { v = $7 + 0; if (v > max) max = v } END { print max + 0 }' "$out_busy/timeline.tsv")"
[ "$max_rss_busy" -gt 32768 ] \
    || { echo "FAIL: expected a sampled RssAnon > 32768 kB after busy-card recovery; got max=$max_rss_busy kB"; fail=1; }

# --- regression: a SHORT-LIVED helper that is NOT a direct child of the
# guard (so the depth gate above cannot exclude it, unlike bench-guard's
# poll `sleep`) must never overwrite the audit line once the real bench has
# already been confirmed (llama.cpp-gvu7 review, round 2, finding 1).
# bench-guard's postflight runs kernel_log() -- and thus --journalctl-cmd --
# strictly AFTER the wrapped command exits, at depth 2 under the guard (a
# command substitution's subshell, then the command itself). An 0.8s fake
# journalctl stands in for the ~10ms real one (measured by the reviewer),
# giving the 0.5s-tick confirmation window a real chance to be fooled if
# two-tick confirmation alone were the only protection -- it is not: a
# candidate is only ever committed as a REPLACEMENT for an existing
# bench_pid, and by the time journalctl runs the real bench has already
# been confirmed and is dead, so this exercises exactly that replacement
# path. ---

out_helper="$T/out-helper-race"
mk_tree 0 0; mk_meminfo 3000000
bench="$(mk_fake_bench_grow 40.0)"
export FAKE_BENCH_PIDFILE="$T/fake-bench-helper.pid"
"$CAPTURE" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --journalctl-cmd "/usr/bin/sleep 0.8" --max-wait 1 \
    --out "$out_helper" -- "$bench" || { echo "FAIL: helper-race run failed"; fail=1; }
unset FAKE_BENCH_PIDFILE

[ -s "$T/fake-bench-helper.pid" ] || { echo "FAIL: fake bench never wrote its own pid (helper-race)"; fail=1; }
want_helper_pid="$(cat "$T/fake-bench-helper.pid")"
grep -q "^bench_pid=$want_helper_pid comm=" "$out_helper/host.txt" \
    || { echo "FAIL: expected bench_pid=$want_helper_pid in host.txt after the postflight helper race (got: $(grep '^bench_pid=' "$out_helper/host.txt" 2>/dev/null))"; fail=1; }
grep -qE '^bench_pid=[0-9]+ comm=sleep ' "$out_helper/host.txt" \
    && { echo "FAIL: host.txt names the postflight journalctl helper, not the bench: $(grep '^bench_pid=' "$out_helper/host.txt")"; fail=1; }

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

# --- regression: a SETUP-ONLY failure (no --sysfs-card and no usable
# ONEAPI_DEVICE_SELECTOR) must exit 3 WITHOUT touching a populated --out dir
# at all (llama.cpp-gvu7 quality review round 2, finding 1 -- a regression
# from round 1's own finding 7, which grouped the four resets together but
# in doing so accidentally moved them to run BEFORE card derivation: an
# earlier version of this script would delete bench.log/mode.txt and blank
# host.txt/timeline.tsv from a completed prior run in a reused --out dir,
# even though card derivation then failed and nothing was ever launched).
# Unlike the high-Shmem refusal case below (a bench-guard.sh PREFLIGHT
# refusal, reached only after --sysfs-card is already known-good), this
# exercises the capture script's OWN card-derivation failure, which must
# happen -- and must exit 3 -- before the reset block is ever reached. ---

out_setup_fail="$T/out-setup-fail"
mk_tree 0 0; mk_meminfo 3000000
bench="$(mk_fake_bench 40.0 "$MK_FAKE_BENCH_FAST_SECONDS")"
run_capture "$out_setup_fail" -- "$bench" || { echo "FAIL: setup-fail-regression seed run failed"; fail=1; }
for f in mode.txt bench.log timeline.tsv host.txt; do
    [ -s "$out_setup_fail/$f" ] || { echo "FAIL: seed run for setup-fail regression left $f missing/empty"; fail=1; }
done
before_mode="$(cat "$out_setup_fail/mode.txt")"
before_log="$(cat "$out_setup_fail/bench.log")"
before_timeline="$(cat "$out_setup_fail/timeline.tsv")"
before_host="$(cat "$out_setup_fail/host.txt")"

setup_rc=0
setup_out="$( ( unset ONEAPI_DEVICE_SELECTOR; "$CAPTURE" --out "$out_setup_fail" -- "$bench" ) 2>&1 )" || setup_rc=$?
[ "$setup_rc" -eq 3 ] \
    || { echo "FAIL: expected a setup-only failure (no --sysfs-card, no selector) to exit 3, got $setup_rc"; fail=1; }
# M2 (spec review round 2): confirm the selector-shape refusal actually goes
# through refuse() (F4/F8, previous round) rather than some other message
# shape -- a bare "exit 3" check above would pass even if this specific
# refusal regressed back to its own bespoke prefix.
echo "$setup_out" | grep -q "sycl-decode-mode-capture: REFUSED:" \
    || { echo "FAIL: setup-only failure (no selector) must use the unified refuse() prefix 'sycl-decode-mode-capture: REFUSED:' (got: $setup_out)"; fail=1; }

[ "$(cat "$out_setup_fail/mode.txt")" = "$before_mode" ] \
    || { echo "FAIL: setup-only failure altered mode.txt in a populated --out dir"; fail=1; }
[ "$(cat "$out_setup_fail/bench.log")" = "$before_log" ] \
    || { echo "FAIL: setup-only failure altered bench.log in a populated --out dir"; fail=1; }
[ "$(cat "$out_setup_fail/timeline.tsv")" = "$before_timeline" ] \
    || { echo "FAIL: setup-only failure altered timeline.tsv in a populated --out dir"; fail=1; }
[ "$(cat "$out_setup_fail/host.txt")" = "$before_host" ] \
    || { echo "FAIL: setup-only failure altered host.txt in a populated --out dir"; fail=1; }

# --- exit status mirrors the underlying run (0 on a clean run) ---

mk_tree 0 0; mk_meminfo 3000000
bench="$(mk_fake_bench 40.0 "$MK_FAKE_BENCH_FAST_SECONDS")"
expect_status 0 "clean run must exit 0" -- run_capture "$T/out-rc" -- "$bench"

# --- refusal: high Shmem (zero tmpfs) must propagate bench-guard's exit 3
# and must NOT leave a stale mode.txt/bench.log from an EARLIER successful
# run in the same --out dir (llama.cpp-gvu7 quality review, finding 1).
# Reuses out_slow, which the very first test case above already populated
# with a real mode.txt (tg128=28.0 mode=slow) and bench.log (a VALID
# header) -- a fresh, never-before-used --out dir could not catch this
# regression, since bench-guard.sh's own refuse() path never touches --log
# at all: without an explicit reset, mode.txt and bench.log would otherwise
# survive completely unchanged from the earlier run, and this script's own
# refusal message would point the reader at the now-stale bench.log. The
# bench command itself is irrelevant here (refused at preflight, never
# executed), so it uses the shortest fixture. ---

mk_tree 0 0; mk_meminfo 30000000
bench="$(mk_fake_bench 40.0 "$MK_FAKE_BENCH_FAST_SECONDS")"
expect_status 3 "high-Shmem refusal into a reused --out dir must propagate as exit 3" -- run_capture "$out_slow" -- "$bench"
[ ! -f "$out_slow/mode.txt" ] || { echo "FAIL: refusal into a reused --out dir must remove the PRIOR run's mode.txt, not just skip writing a new one"; fail=1; }
[ ! -f "$out_slow/bench.log" ] || { echo "FAIL: refusal into a reused --out dir must remove the PRIOR run's bench.log"; fail=1; }

[ "$fail" -eq 0 ] && echo "OK: sycl-decode-mode-capture" || exit 1
