#!/usr/bin/env bash
# Unit tests for scripts/sycl-prefill-scaling.sh, against a fake sysfs tree
# (mirrors tests/test-bench-guard.sh's mk_tree/mk_meminfo pattern -- the
# script wraps bench-guard.sh as a child, exactly like
# tests/test-sycl-decode-mode-capture.sh's own suite does for
# sycl-decode-mode-capture.sh), a fake `llama-bench` that prints canned
# markdown table rows instead of running any GPU work, and (for the
# selector-propagation case) a stub bench-guard that records what it was
# invoked with instead of touching any real sysfs. Pure bash, no GPU, safe
# at any parallelism.
#
# What this suite proves, and why each case is here:
#   - the parser extracts pp128/pp512/pp1024/pp2048 from a real-shaped
#     markdown table (row values below are the literal collapse numbers
#     recorded in docs/backend/sycl-perf-baselines.md's 2026-09-04 scaling
#     table for B70 Mistral 7B Q4_0 -- this is not an invented case, it is
#     the exact regression the gate exists to keep visible);
#   - the ratio1024=pp1024/pp512 verdict flips at EXACTLY 0.9 -- 0.9 itself
#     must PASS (the acceptance criterion is "< 0.9" fails, so "== 0.9" does
#     not), one unit below must FAIL;
#   - a bench-guard preflight refusal (throttled card) on one pair is
#     reported as an error distinct from a computed-but-failing ratio, and
#     takes the script to its own error exit code, never silently to 0;
#   - --only filters to the requested pair(s) and no others, and an
#     --only token that names no such pair is a loud usage error, never a
#     silent empty "OK";
#   - the pp128/pp512 intercept (fixed per-decode cost) is computed and
#     printed, from the closed-form two-point line pp128/pp512 imply;
#   - ONEAPI_DEVICE_SELECTOR reaches bench-guard's OWN environment, set
#     per pair (level_zero:0 for B70, level_zero:1 for B50), via a stub
#     guard that records what it was invoked with (rev-y3z0-spec-1
#     finding 1: it used to be set only on the wrapped bench, so the real
#     guard could never derive a card at all);
#   - a wrapped bench that exits non-zero after printing a healthy table,
#     or a bench-guard log stamped SUSPECT (a kernel GPU fault mid-run),
#     is reported as an unmeasured ERROR, never a computed PASS/FAIL
#     (rev-y3z0-spec-1 finding 2);
#   - a genuine ratio<0.9 FAIL on one pair outranks an unrelated
#     unmeasurable pair in the same run: the mixed case exits 1, not 2,
#     and names both (rev-y3z0-spec-1 finding 4);
#   - the parser anchors on an EXACT cell match, not a substring: fed a
#     real-shaped table (fa column, `±` spread, ngl=-1, log noise, rows in
#     non-canonical order) plus a decoy row whose MODEL field contains
#     "pp128" as a substring, the correct pp128 value is still extracted,
#     never the decoy's (rev-y3z0-spec-1 finding 5).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCALING="$ROOT_DIR/scripts/sycl-prefill-scaling.sh"
GUARD="$ROOT_DIR/scripts/bench-guard.sh"
# SCALING ships in the SAME commit as this test (RED before the script
# exists is expected and is the point -- see the commit body's captured RED
# run -- but a checkout that has since lost it is a defect, not a skip).
[ -x "$SCALING" ] || { echo "FAIL: $SCALING is missing or not executable -- it ships in the same commit as this test" >&2; exit 1; }
[ -x "$GUARD" ] || { echo "SKIP: bench-guard.sh not present"; exit 77; }

export LC_NUMERIC=C

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0

# --- fake sysfs / meminfo fixtures, mirrors test-bench-guard.sh ---
mk_tree() { # $1=throttle $2=act_freq
    local d="$T/sys/class/drm/card9/device/tile0/gt0/freq0"
    mkdir -p "$d/throttle"
    echo "$1" > "$d/throttle/status"
    echo "$2" > "$d/act_freq"
}
mk_meminfo() { printf 'MemAvailable: 190000000 kB\nShmem: %s kB\n' "$1" > "$T/meminfo"; }
mk_tree 0 0
mk_meminfo 3000000

# GUARD_HOOKS: the fixture args every invocation below forwards through
# sycl-prefill-scaling.sh to bench-guard.sh, so the real guard's preflight
# runs against the fake tree instead of this host's actual hardware.
# --df-cmd true: tmpfs usage 0 kB, so the fake Shmem above is never clamped
# or contested by this host's real tmpfs (test-bench-guard.sh's own
# hermeticity guard, same reasoning). --journalctl-cmd true: a clean "no
# kernel fault" answer, matching tests/test-sycl-decode-mode-capture.sh:225
# -- without it, every invocation below shells out to this host's REAL
# `journalctl -k`, which is harmless only by accident and becomes a live
# flake risk the moment a run's own postflight check starts to matter
# (rev-y3z0-spec-1 finding 6).
GUARD_HOOKS=(--sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd false --df-cmd true --journalctl-cmd true --max-wait 1)

# mk_fake_bench: writes an executable at $1 that ignores every argument and
# prints a markdown table with the four rows below verbatim on stdout, then
# exits 0. Table shape (header, alignment row, four `test` rows) mirrors
# artifacts/task18-parser-fixtures/b70-mistral-good.txt, reconstructed from
# tools/llama-bench/llama-bench.cpp's markdown_printer the same way that
# fixture's own README documents -- the model/backend/ngl columns are
# plausible filler, not asserted on; only the `test` and `t/s` cells matter
# to the parser under test. A table with the FULL real column set (fa,
# `±` spread, ngl=-1, log noise, shuffled rows, a substring-colliding decoy
# row) is exercised separately below, not by this helper.
mk_fake_bench() {
    local path="$1" pp128="$2" pp512="$3" pp1024="$4" pp2048="$5"
    mk_fake_bench_rc "$path" "$pp128" "$pp512" "$pp1024" "$pp2048" 0
}

# mk_fake_bench_rc: like mk_fake_bench, but the generated script exits
# $6 instead of always 0 -- lets a case print a fully healthy table and
# still fail as if the bench crashed/was killed right after (rev-y3z0-
# spec-1 finding 2).
mk_fake_bench_rc() {
    local path="$1" pp128="$2" pp512="$3" pp1024="$4" pp2048="$5" exitcode="$6"
    cat > "$path" <<EOF
#!/usr/bin/env bash
cat <<'TABLE'
| model                          |       size |     params | backend    | ngl |             test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ----------------: | -------------------: |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |             pp128 |      ${pp128} |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |             pp512 |      ${pp512} |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |            pp1024 |     ${pp1024} |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |            pp2048 |     ${pp2048} |

build: df51c5130 (7412)
TABLE
exit ${exitcode}
EOF
    chmod +x "$path"
}

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

# --- Case 1: the real 2026-09-04 collapse numbers (B70 Mistral 7B Q4_0),
# docs/backend/sycl-perf-baselines.md line 156: pp128=1315 pp512=3320
# pp1024=1437 pp2048=1474. ratio1024 = 1437/3320 = 0.4328..., far under the
# 0.9 floor -- this is the exact regression the gate exists to keep visible,
# not a synthetic number. Single pair via --only, and the printed table must
# carry all four values plus the low ratio.
BENCH1="$T/fake-bench-collapse.sh"
mk_fake_bench "$BENCH1" "1315.00" "3320.00" "1437.00" "1474.00"
out="$("$SCALING" --bench "$BENCH1" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: collapse case must exit 1 (verdict fail), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -q "1315.00" || { echo "FAIL: pp128 value missing from table (got: $out)"; fail=1; }
echo "$out" | grep -q "3320.00" || { echo "FAIL: pp512 value missing from table (got: $out)"; fail=1; }
echo "$out" | grep -q "1437.00" || { echo "FAIL: pp1024 value missing from table (got: $out)"; fail=1; }
echo "$out" | grep -q "1474.00" || { echo "FAIL: pp2048 value missing from table (got: $out)"; fail=1; }
echo "$out" | grep -qE 'ratio1024=0\.43' || { echo "FAIL: expected ratio1024=0.43... in output (got: $out)"; fail=1; }
echo "$out" | grep -qi "FAIL" || { echo "FAIL: overall verdict must mention FAIL (got: $out)"; fail=1; }

# --- Case 2: ratio1024 EXACTLY 0.9 must PASS (acceptance criterion is
# "< 0.9" fails; 0.9 itself does not qualify). pp512=1000.00, pp1024=900.00.
BENCH2="$T/fake-bench-boundary-pass.sh"
mk_fake_bench "$BENCH2" "1000.00" "1000.00" "900.00" "850.00"
out="$("$SCALING" --bench "$BENCH2" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: ratio1024==0.9 must PASS (exit 0), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qE 'ratio1024=0\.90+([^0-9]|$)' || { echo "FAIL: expected ratio1024=0.900 in output (got: $out)"; fail=1; }

# --- Case 3: one unit of t/s below the boundary (pp1024=899.99) must FAIL.
BENCH3="$T/fake-bench-boundary-fail.sh"
mk_fake_bench "$BENCH3" "1000.00" "1000.00" "899.99" "850.00"
expect_status 1 "ratio1024 just under 0.9 must FAIL" -- \
    "$SCALING" --bench "$BENCH3" --only mistral,b70 "${GUARD_HOOKS[@]}"

# --- Case 4: --only restricts to the requested pair(s) and no others --
# a fake bench that always reports the SAME healthy numbers regardless of
# model/card, so if a second, unrequested pair were run, its row would
# appear too. Assert exactly one data row block (one "ratio1024=" line) is
# printed for a single --only.
BENCH4="$T/fake-bench-healthy.sh"
mk_fake_bench "$BENCH4" "3200.00" "3300.00" "3100.00" "3000.00"
out="$("$SCALING" --bench "$BENCH4" --only gptoss,b50 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: single healthy pair must PASS (exit 0), got $rc. Output:
$out"; fail=1; }
n_ratio_lines="$(echo "$out" | grep -c 'ratio1024=' || true)"
[ "$n_ratio_lines" -eq 1 ] || { echo "FAIL: --only gptoss,b50 must print exactly one ratio1024 line, got $n_ratio_lines. Output:
$out"; fail=1; }
echo "$out" | grep -qi "gptoss\|GPT-OSS" || { echo "FAIL: expected the GPT-OSS row to be present (got: $out)"; fail=1; }
echo "$out" | grep -qi "mistral\|gemma4" && { echo "FAIL: --only must exclude other models (got: $out)"; fail=1; }

# --- Case 5: a bench-guard preflight refusal (throttled card) on the
# requested pair must be reported as an ERROR distinct from a computed
# ratio, and the script's own exit code must not be 0 or the plain verdict-
# fail 1 -- a silent 0 here would read as "the collapse gate passed" on a
# pair that was never actually measured.
mk_tree 1 0   # throttle=1 -> bench-guard refuses at preflight
out="$("$SCALING" --bench "$BENCH4" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -ge 2 ] || { echo "FAIL: a bench-guard refusal must not exit 0 or 1, got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "refus\|error" || { echo "FAIL: refusal must be reported (got: $out)"; fail=1; }
mk_tree 0 0   # restore clean fixture for any cases added below this line

# --- Case 6: the pp128/pp512 intercept (fixed per-decode cost) is computed
# from the closed-form two points (128, 128/pp128) and (512, 512/pp512):
#   t128 = 128/1000.00 = 0.128000 s,  t512 = 512/1000.00 = 0.512000 s
#   slope = (t512 - t128) / (512 - 128) = 0.384000 / 384 = 0.0010000 s/tok
#   intercept = t128 - slope*128 = 0.128000 - 0.128000 = 0.000000 s
# A flat-rate input (pp128 == pp512) has NO fixed cost by construction, so
# intercept_ms must read ~0 -- this is a positive control on the formula
# itself (a bug that always printed a nonzero constant would be caught
# here, not just an ordinary case where the true value can't be told apart
# from a bug printing zero).
out="$("$SCALING" --bench "$BENCH2" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
echo "$out" | grep -qE 'intercept_ms=-?0\.0*([^0-9]|$)' || { echo "FAIL: expected intercept_ms=~0 for a flat-rate input (got: $out)"; fail=1; }

# A second intercept case with a genuine fixed cost: pp128=1000.00 (t128=
# 0.128s), pp512=4000.00 (t512=0.128s) -- SAME total time at both sizes is
# only possible if the fixed cost absorbs the entire extra 384 tokens'
# worth of time, i.e. slope=0 and intercept=0.128s=128ms exactly.
BENCH5="$T/fake-bench-intercept.sh"
mk_fake_bench "$BENCH5" "1000.00" "4000.00" "3900.00" "3800.00"
out="$("$SCALING" --bench "$BENCH5" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
echo "$out" | grep -qE 'intercept_ms=128\.0*([^0-9]|$)' || { echo "FAIL: expected intercept_ms=128.0 (got: $out)"; fail=1; }

# --- Case 7 (rev-y3z0-spec-1 finding 1): ONEAPI_DEVICE_SELECTOR must reach
# bench-guard's OWN environment, set per pair -- level_zero:0 for B70,
# level_zero:1 for B50 -- not only the wrapped bench's. A stub guard
# records what selector it was invoked with (and the wrapped command's own
# argv) to STUB_GUARD_AUDIT, then runs the wrapped command itself and
# mirrors its exit code; it owns no preflight of its own, so none of
# GUARD_HOOKS is needed for this case. Two pairs on the same model, two
# different cards, one invocation: both audit lines must show the RIGHT
# selector, and the audit log must not still be empty (a positive control
# on the stub itself: if nothing was ever appended, every assertion below
# would vacuously fail rather than vacuously pass, but this line makes
# that failure mode explicit).
STUB_GUARD="$T/stub-guard.sh"
cat > "$STUB_GUARD" <<'EOF'
#!/usr/bin/env bash
# Records "<ONEAPI_DEVICE_SELECTOR or <unset>> <wrapped command argv>" to
# $STUB_GUARD_AUDIT, then behaves like a minimal bench-guard: consumes
# --log FILE (writing a VALID header + the wrapped command's output into
# it, mirroring the real bench-guard.sh's --log contract closely enough
# for sycl-prefill-scaling.sh's own header check to accept it) and any of
# the real guard's test-hook flags (accepted and ignored -- this stub
# owns no preflight of its own), runs the wrapped command, and mirrors
# its exit status.
set -euo pipefail
LOG=""
while [ $# -gt 0 ]; do case "$1" in
    --log) LOG="$2"; shift 2;;
    --sysfs-card|--meminfo|--pgrep-cmd|--df-cmd|--journalctl-cmd|--max-wait|--budget) shift 2;;
    --) shift; break;;
    *) shift;;
esac; done
: "${STUB_GUARD_AUDIT:?STUB_GUARD_AUDIT must be set}"
printf '%s %s\n' "${ONEAPI_DEVICE_SELECTOR:-<unset>}" "$*" >> "$STUB_GUARD_AUDIT"
rc=0
if [ -n "$LOG" ]; then
    { echo "# bench-guard: VALID (stub)"; "$@"; } > "$LOG" 2>&1 || rc=$?
else
    "$@" || rc=$?
fi
exit "$rc"
EOF
chmod +x "$STUB_GUARD"

AUDIT="$T/stub-audit.log"
: > "$AUDIT"
out="$(STUB_GUARD_AUDIT="$AUDIT" "$SCALING" --bench "$BENCH4" --guard "$STUB_GUARD" --only mistral,b70 --only mistral,b50 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: selector-propagation case must PASS (exit 0) via the stub guard, got $rc. Output:
$out"; fail=1; }
[ -s "$AUDIT" ] || { echo "FAIL: stub guard audit log is empty -- the stub was never invoked, so the assertions below would pass vacuously"; fail=1; }
grep -qE '^level_zero:0 .*mistral-7b-v0\.1\.Q4_0\.gguf' "$AUDIT" || { echo "FAIL: expected a level_zero:0 (B70) audit line for mistral (got: $(cat "$AUDIT"))"; fail=1; }
grep -qE '^level_zero:1 .*mistral-7b-v0\.1\.Q4_0\.gguf' "$AUDIT" || { echo "FAIL: expected a level_zero:1 (B50) audit line for mistral (got: $(cat "$AUDIT"))"; fail=1; }
grep -q '<unset>' "$AUDIT" && { echo "FAIL: ONEAPI_DEVICE_SELECTOR must never reach the guard unset (got: $(cat "$AUDIT"))"; fail=1; }

# --- Case 8 (rev-y3z0-spec-1 finding 2, part A): a wrapped bench that
# prints a FULLY HEALTHY table and then exits non-zero (crashed/killed
# right after) must be reported as an unmeasured ERROR, never a computed
# PASS -- the numbers it printed cannot be trusted just because they
# parse.
BENCH_CRASH="$T/fake-bench-crash.sh"
mk_fake_bench_rc "$BENCH_CRASH" "1000.00" "1000.00" "990.00" "980.00" 134
out="$("$SCALING" --bench "$BENCH_CRASH" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a bench that exits 134 after a healthy table must exit 2 (ERROR), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "ERROR" || { echo "FAIL: expected an ERROR row/summary (got: $out)"; fail=1; }
echo "$out" | grep -qi "PASS" && { echo "FAIL: a crashed run must never report PASS (got: $out)"; fail=1; }
echo "$out" | grep -q "990.00" && { echo "FAIL: a crashed run's numbers must not be printed as measured (got: $out)"; fail=1; }
# rev-y3z0-spec-2 finding N4: a non-zero BENCH exit gets its own distinct
# label (ERROR:bench-rc=<n>), never the "the guard's postflight flagged
# this" label a SUSPECT run gets below -- the two are different failure
# modes with different next steps for whoever reads the table.
echo "$out" | grep -q "ERROR:bench-rc=134" || { echo "FAIL: expected the distinct label ERROR:bench-rc=134 for a non-zero bench exit (got: $out)"; fail=1; }
echo "$out" | grep -q "ERROR:guard-not-valid" && { echo "FAIL: a non-zero bench exit must use ERROR:bench-rc=, not ERROR:guard-not-valid (got: $out)"; fail=1; }

# --- Case 8 (rev-y3z0-spec-1 finding 2, part B): a bench-guard log
# stamped SUSPECT (here: a fake journalctl reporting a GT reset, i.e. a
# kernel GPU fault during the run) must be reported as an unmeasured
# ERROR even though the wrapped bench itself printed a healthy table and
# exited 0 -- CLAUDE.md is explicit that numbers from a run invalidated by
# a GPU fault are not usable, gate or no gate.
out="$("$SCALING" --bench "$BENCH2" --only mistral,b70 "${GUARD_HOOKS[@]}" \
    --journalctl-cmd "echo kernel: xe 0000:03:00.0: GT reset triggered" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a SUSPECT (GT-reset) run must exit 2 (ERROR), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "ERROR" || { echo "FAIL: expected an ERROR row/summary for the SUSPECT run (got: $out)"; fail=1; }
echo "$out" | grep -qi "PASS" && { echo "FAIL: a SUSPECT run must never report PASS (got: $out)"; fail=1; }
# rev-y3z0-spec-2 finding N4: the guard's own postflight (VALID-stamp
# missing, here from a fake GT-reset journalctl) gets the distinct
# ERROR:guard-not-valid label, never ERROR:bench-rc= -- the wrapped bench
# itself exited 0 here (its healthy table is the reason the "must never
# report PASS" assertion above matters at all), so a bench-rc label would
# misreport WHICH layer flagged the run.
echo "$out" | grep -q "ERROR:guard-not-valid" || { echo "FAIL: expected the distinct label ERROR:guard-not-valid for a SUSPECT-stamped run (got: $out)"; fail=1; }
echo "$out" | grep -q "ERROR:bench-rc=" && { echo "FAIL: a SUSPECT run (bench itself exited 0) must use ERROR:guard-not-valid, not ERROR:bench-rc= (got: $out)"; fail=1; }

# --- Case 8 (rev-y3z0-spec-2 finding N3): a positive control on the
# exact-token VALID match. A second stub guard stamps its --log header
# "# bench-guard: VALIDATED (should NOT count as VALID)" -- a real
# bench-guard.sh never emits this (its verdict_line is always exactly
# "VALID" or "SUSPECT:<reasons>"), but the OLD glob match
# (`"# bench-guard: VALID"*`) would have accepted this header as a real
# VALID stamp and gone on to compute an ordinary verdict from the healthy
# table underneath it. The fixed exact-token regex must reject it as
# ERROR:guard-not-valid instead.
STUB_GUARD_VALIDATED="$T/stub-guard-validated.sh"
cat > "$STUB_GUARD_VALIDATED" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
LOG=""
while [ $# -gt 0 ]; do case "$1" in
    --log) LOG="$2"; shift 2;;
    --sysfs-card|--meminfo|--pgrep-cmd|--df-cmd|--journalctl-cmd|--max-wait|--budget) shift 2;;
    --) shift; break;;
    *) shift;;
esac; done
rc=0
if [ -n "$LOG" ]; then
    { echo "# bench-guard: VALIDATED (should NOT count as VALID)"; "$@"; } > "$LOG" 2>&1 || rc=$?
else
    "$@" || rc=$?
fi
exit "$rc"
EOF
chmod +x "$STUB_GUARD_VALIDATED"
out="$("$SCALING" --bench "$BENCH2" --guard "$STUB_GUARD_VALIDATED" --only mistral,b70 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a 'VALIDATED' (not 'VALID') header must exit 2 (ERROR), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -q "ERROR:guard-not-valid" || { echo "FAIL: a 'VALIDATED' header must be rejected as ERROR:guard-not-valid, not accepted as VALID (got: $out)"; fail=1; }
echo "$out" | grep -qi "PASS" && { echo "FAIL: a 'VALIDATED' header must never be accepted as a real VALID stamp (got: $out)"; fail=1; }


# --- Case 9 (rev-y3z0-spec-1 finding 4): precedence. One pair genuinely
# FAILs (ratio<0.9, from the real collapse numbers), the other cannot be
# measured at all (the fake bench exits 77 for any model path other than
# mistral's) -- the mixed run must exit 1 (a real regression outranks an
# unrelated measurement gap), and the summary must name both conditions,
# not just one.
BENCH_MIXED="$T/fake-bench-mixed.sh"
cat > "$BENCH_MIXED" <<'EOF'
#!/usr/bin/env bash
m=""
while [ $# -gt 0 ]; do case "$1" in -m) m="$2"; shift 2;; *) shift;; esac; done
case "$m" in
    *mistral*)
        cat <<'TABLE'
| model         |       size |     params | backend    | ngl |    test |         t/s |
| ------------- | ---------: | ---------: | ---------- | --: | ------: | -----------: |
| llama 7B Q4_0 |   3.83 GiB |     7.24 B | SYCL       |  99 |   pp128 |      1315.00 |
| llama 7B Q4_0 |   3.83 GiB |     7.24 B | SYCL       |  99 |   pp512 |      3320.00 |
| llama 7B Q4_0 |   3.83 GiB |     7.24 B | SYCL       |  99 |  pp1024 |      1437.00 |
| llama 7B Q4_0 |   3.83 GiB |     7.24 B | SYCL       |  99 |  pp2048 |      1474.00 |
TABLE
        exit 0
        ;;
    *)
        exit 77
        ;;
esac
EOF
chmod +x "$BENCH_MIXED"
out="$("$SCALING" --bench "$BENCH_MIXED" --only mistral,b70 --only gptoss,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: FAIL+unmeasurable mixed case must exit 1 (FAIL outranks ERROR), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "FAIL" || { echo "FAIL: expected the mistral row/summary to mention FAIL (got: $out)"; fail=1; }
echo "$out" | grep -qi "ERROR" || { echo "FAIL: expected the gptoss row to be reported as ERROR (got: $out)"; fail=1; }
echo "$out" | grep -qi "additionally\|also" || { echo "FAIL: the summary must name BOTH the FAIL and the unmeasured pair, not just one (got: $out)"; fail=1; }

# --- Case 10 (rev-y3z0-spec-1 finding 3): an --only token that names no
# such pair (wrong case here: "B70" instead of "b70") must be a loud usage
# error naming the valid keys, never a silently empty "OK" table.
out="$("$SCALING" --bench "$BENCH2" --only mistral,B70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a typo'd --only must exit 2 (usage error), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "valid keys" || { echo "FAIL: the error must name the valid --only keys (got: $out)"; fail=1; }
echo "$out" | grep -qi "^OK" && { echo "FAIL: a typo'd --only must never read as OK (got: $out)"; fail=1; }
# rev-y3z0-spec-2 finding N2: the table HEADER (its last field is the bare
# word "status" -- every data/error row instead ends "PASS", "FAIL(...)",
# or an "ERROR:..." label) must never print at all. Without this, a
# mutant that deletes the UP-FRONT --only validation and relies solely on
# the post-loop "any_measured==0" fallback would still pass every other
# assertion in this case: the header is printed unconditionally BEFORE
# the main loop runs, so removing only the up-front check still yields
# rc=2 and the same "valid keys" message from the fallback, just with the
# header already on stdout by the time it fires. This assertion is what
# actually pins "the up-front check runs", not merely "some check runs
# eventually" -- demonstrated with a scratch mutant (up-front validation
# loop commented out) in this round's commit body.
echo "$out" | grep -qw "status" && { echo "FAIL: the table header must never print for a rejected --only -- this means the UP-FRONT --only validation did not run before the header printf, and only the post-loop fallback caught it (got: $out)"; fail=1; }

# --- Case 11 (rev-y3z0-spec-1 finding 5): a REAL-shaped table -- fa
# column, `±` spread, ngl=-1, surrounding log noise exactly like a real
# capture (artifacts/task18-parser-fixtures/b70-mistral-good.txt), rows in
# non-canonical order, PLUS a decoy row whose MODEL field contains "pp128"
# as a substring (test cell "warmup", a implausible-looking value
# 9999.99) placed immediately after the real pp128 row. An exact-cell-
# match parser must return the real pp128 value (1315.22); a substring-
# based mutant (`index(s, want) > 0` in place of `s == want`) would also
# match the decoy row, and since parse_cell takes the LAST matching row,
# would return the decoy's bogus 9999.99 instead -- this is a shipped,
# reproducible pin of the property the prior round only checked by hand
# with a throwaway scratch mutation.
BENCH_REALISTIC="$T/fake-bench-realistic.sh"
cat > "$BENCH_REALISTIC" <<'EOF'
#!/usr/bin/env bash
cat <<'TABLE'
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
llama_model_loader: loaded meta data with 24 key-value pairs and 291 tensors from /models/mistral-7b-v0.1.Q4_0.gguf
llama_prepare_model_devices: using device SYCL0 (Intel(R) Arc(TM) Pro B70 Graphics) (unknown id) - 32602 MiB free
| model                          |       size |     params | backend    | ngl | fa |             test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ----------------: | -------------------: |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  -1 |  1 |             pp512 |      3320.11 ± 15.00 |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  -1 |  1 |            pp1024 |      1437.33 ± 12.00 |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  -1 |  1 |             pp128 |      1315.22 ±  5.00 |
| llama-pp128collide 7B Q4_0     |   3.83 GiB |     7.24 B | SYCL       |  -1 |  1 |            warmup |      9999.99 ±  1.00 |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  -1 |  1 |            pp2048 |      1474.44 ± 20.00 |

build: df51c5130 (7412)
TABLE
EOF
chmod +x "$BENCH_REALISTIC"
out="$("$SCALING" --bench "$BENCH_REALISTIC" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: realistic-table case must exit 1 (its ratio1024 is the same collapse), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -q "1315.22" || { echo "FAIL: expected the REAL pp128 value 1315.22 despite the substring-colliding decoy row (got: $out)"; fail=1; }
echo "$out" | grep -q "3320.11" || { echo "FAIL: pp512 value missing (got: $out)"; fail=1; }
echo "$out" | grep -q "1437.33" || { echo "FAIL: pp1024 value missing (got: $out)"; fail=1; }
echo "$out" | grep -q "1474.44" || { echo "FAIL: pp2048 value missing (got: $out)"; fail=1; }
echo "$out" | grep -q "9999.99" && { echo "FAIL: the decoy row's bogus value must never leak into the parsed table (got: $out)"; fail=1; }

[ "$fail" -eq 0 ] && echo "OK: prefill scaling parser and ratio verdict" || exit 1
